import copy
import datetime
import sqlalchemy
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from keystone.common import provider_api
from keystone.common import sql
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import sql_model as model
from keystone.identity.shadow_backends import base
def add_user_to_group_expires(self, user_id, group_id):

    def get_federated_user():
        with sql.session_for_read() as session:
            query = session.query(model.FederatedUser)
            query = query.filter_by(user_id=user_id)
            user = query.first()
            if not user:
                raise exception.UserNotFound(user_id=user_id)
            return user
    with sql.session_for_write() as session:
        user = get_federated_user()
        query = session.query(model.ExpiringUserGroupMembership)
        query = query.filter_by(user_id=user_id)
        query = query.filter_by(group_id=group_id)
        membership = query.first()
        if membership:
            membership.last_verified = datetime.datetime.utcnow()
        else:
            session.add(model.ExpiringUserGroupMembership(user_id=user_id, group_id=group_id, idp_id=user.idp_id, last_verified=datetime.datetime.utcnow()))