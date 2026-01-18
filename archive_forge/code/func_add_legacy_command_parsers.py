import os
import sys
import time
from alembic import command as alembic_command
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_log import log as logging
from oslo_utils import encodeutils
from glance.common import config
from glance.common import exception
from glance import context
from glance.db import migration as db_migration
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata
from glance.i18n import _
def add_legacy_command_parsers(command_object, subparsers):
    legacy_command_object = DbLegacyCommands(command_object)
    parser = subparsers.add_parser('db_version')
    parser.set_defaults(action_fn=legacy_command_object.version)
    parser.set_defaults(action='db_version')
    parser = subparsers.add_parser('db_upgrade')
    parser.set_defaults(action_fn=legacy_command_object.upgrade)
    parser.add_argument('version', nargs='?')
    parser.set_defaults(action='db_upgrade')
    parser = subparsers.add_parser('db_version_control')
    parser.set_defaults(action_fn=legacy_command_object.version_control)
    parser.add_argument('version', nargs='?')
    parser.set_defaults(action='db_version_control')
    parser = subparsers.add_parser('db_sync')
    parser.set_defaults(action_fn=legacy_command_object.sync)
    parser.add_argument('version', nargs='?')
    parser.set_defaults(action='db_sync')
    parser = subparsers.add_parser('db_expand')
    parser.set_defaults(action_fn=legacy_command_object.expand)
    parser.set_defaults(action='db_expand')
    parser = subparsers.add_parser('db_contract')
    parser.set_defaults(action_fn=legacy_command_object.contract)
    parser.set_defaults(action='db_contract')
    parser = subparsers.add_parser('db_migrate')
    parser.set_defaults(action_fn=legacy_command_object.migrate)
    parser.set_defaults(action='db_migrate')
    parser = subparsers.add_parser('db_check')
    parser.set_defaults(action_fn=legacy_command_object.check)
    parser.set_defaults(action='db_check')
    parser = subparsers.add_parser('db_load_metadefs')
    parser.set_defaults(action_fn=legacy_command_object.load_metadefs)
    parser.add_argument('path', nargs='?')
    parser.add_argument('merge', nargs='?')
    parser.add_argument('prefer_new', nargs='?')
    parser.add_argument('overwrite', nargs='?')
    parser.set_defaults(action='db_load_metadefs')
    parser = subparsers.add_parser('db_unload_metadefs')
    parser.set_defaults(action_fn=legacy_command_object.unload_metadefs)
    parser.set_defaults(action='db_unload_metadefs')
    parser = subparsers.add_parser('db_export_metadefs')
    parser.set_defaults(action_fn=legacy_command_object.export_metadefs)
    parser.add_argument('path', nargs='?')
    parser.set_defaults(action='db_export_metadefs')