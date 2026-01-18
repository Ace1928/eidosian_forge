import inspect
import os
import sys
import threading
import warnings
from collections import UserDict, defaultdict, deque
from datetime import datetime
from datetime import timezone as datetime_timezone
from operator import attrgetter
from click.exceptions import Exit
from dateutil.parser import isoparse
from kombu import pools
from kombu.clocks import LamportClock
from kombu.common import oid_from
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import starpromise
from celery import platforms, signals
from celery._state import (_announce_app_finalized, _deregister_app, _register_app, _set_current_app, _task_stack,
from celery.exceptions import AlwaysEagerIgnored, ImproperlyConfigured
from celery.loaders import get_loader_cls
from celery.local import PromiseProxy, maybe_evaluate
from celery.utils import abstract
from celery.utils.collections import AttributeDictMixin
from celery.utils.dispatch import Signal
from celery.utils.functional import first, head_from_fun, maybe_list
from celery.utils.imports import gen_task_name, instantiate, symbol_by_name
from celery.utils.log import get_logger
from celery.utils.objects import FallbackContext, mro_lookup
from celery.utils.time import maybe_make_aware, timezone, to_utc
from . import backends, builtins  # noqa
from .annotations import prepare as prepare_annotations
from .autoretry import add_autoretry_behaviour
from .defaults import DEFAULT_SECURITY_DIGEST, find_deprecated_settings
from .registry import TaskRegistry
from .utils import (AppPickler, Settings, _new_key_to_old, _old_key_to_new, _unpickle_app, _unpickle_app_v2, appstr,
def setup_security(self, allowed_serializers=None, key=None, key_password=None, cert=None, store=None, digest=DEFAULT_SECURITY_DIGEST, serializer='json'):
    """Setup the message-signing serializer.

        This will affect all application instances (a global operation).

        Disables untrusted serializers and if configured to use the ``auth``
        serializer will register the ``auth`` serializer with the provided
        settings into the Kombu serializer registry.

        Arguments:
            allowed_serializers (Set[str]): List of serializer names, or
                content_types that should be exempt from being disabled.
            key (str): Name of private key file to use.
                Defaults to the :setting:`security_key` setting.
            key_password (bytes): Password to decrypt the private key.
                Defaults to the :setting:`security_key_password` setting.
            cert (str): Name of certificate file to use.
                Defaults to the :setting:`security_certificate` setting.
            store (str): Directory containing certificates.
                Defaults to the :setting:`security_cert_store` setting.
            digest (str): Digest algorithm used when signing messages.
                Default is ``sha256``.
            serializer (str): Serializer used to encode messages after
                they've been signed.  See :setting:`task_serializer` for
                the serializers supported.  Default is ``json``.
        """
    from celery.security import setup_security
    return setup_security(allowed_serializers, key, key_password, cert, store, digest, serializer, app=self)