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
def db_load_metadefs(engine, metadata_path=None, merge=False, prefer_new=False, overwrite=False):
    meta = MetaData()
    if not merge and (prefer_new or overwrite):
        LOG.error(_LE('To use --prefer_new or --overwrite you need to combine of these options with --merge option.'))
        return
    if prefer_new and overwrite and merge:
        LOG.error(_LE('Please provide no more than one option from this list: --prefer_new, --overwrite'))
        return
    with engine.connect() as conn:
        _populate_metadata(meta, conn, metadata_path, merge, prefer_new, overwrite)