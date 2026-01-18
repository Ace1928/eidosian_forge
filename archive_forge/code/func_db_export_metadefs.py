import json
import os
from os.path import isfile
from os.path import join
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import select
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def db_export_metadefs(engine, metadata_path=None):
    meta = MetaData()
    with engine.connect() as conn:
        _export_data_to_file(meta, conn, metadata_path)