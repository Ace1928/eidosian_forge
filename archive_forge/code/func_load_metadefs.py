import threading
from oslo_config import cfg
from oslo_db import options as db_options
from stevedore import driver
from glance.db.sqlalchemy import api as db_api
def load_metadefs():
    """Read metadefinition files and insert data into the database"""
    return get_backend().db_load_metadefs(engine=db_api.get_engine(), metadata_path=None, merge=False, prefer_new=False, overwrite=False)