import threading
from oslo_config import cfg
from oslo_db import options as db_options
from stevedore import driver
from glance.db.sqlalchemy import api as db_api
def export_metadefs():
    """Export metadefinitions from database to files"""
    return get_backend().db_export_metadefs(engine=db_api.get_engine(), metadata_path=None)