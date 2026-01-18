import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
def assert_quota(context, task_repo, task_id, stores, action_wrapper, enforce_quota_fn, **enforce_kwargs):
    try:
        enforce_quota_fn(context, context.owner, **enforce_kwargs)
    except exception.LimitExceeded as e:
        with excutils.save_and_reraise_exception():
            with action_wrapper as action:
                action.remove_importing_stores(stores)
                if action.image_status == 'importing':
                    action.set_image_attribute(status='queued')
            action_wrapper.drop_lock_for_task()
            task = script_utils.get_task(task_repo, task_id)
            if task is None:
                LOG.error(_LE('Failed to find task %r to update after quota failure'), task_id)
            else:
                task.fail(str(e))
                task_repo.save(task)