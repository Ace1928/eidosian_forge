import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_snapshot(snapshot):
    if snapshot is None:
        return
    result = {rpc_api.SNAPSHOT_ID: snapshot.id, rpc_api.SNAPSHOT_NAME: snapshot.name, rpc_api.SNAPSHOT_STATUS: snapshot.status, rpc_api.SNAPSHOT_STATUS_REASON: snapshot.status_reason, rpc_api.SNAPSHOT_DATA: snapshot.data, rpc_api.SNAPSHOT_CREATION_TIME: heat_timeutils.isotime(snapshot.created_at)}
    return result