import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def register_event_callback(event, resource_type, callbacks):
    """Register each callback with the event.

    :param event: Action being registered
    :type event: keystone.notifications.ACTIONS
    :param resource_type: Type of resource being operated on
    :type resource_type: str
    :param callbacks: Callback items to be registered with event
    :type callbacks: list
    :raises ValueError: If event is not a valid ACTION
    :raises TypeError: If callback is not callable
    """
    if event not in ACTIONS:
        raise ValueError(_('%(event)s is not a valid notification event, must be one of: %(actions)s') % {'event': event, 'actions': ', '.join(ACTIONS)})
    if not hasattr(callbacks, '__iter__'):
        callbacks = [callbacks]
    for callback in callbacks:
        if not callable(callback):
            msg = 'Method not callable: %s' % callback
            tr_msg = _('Method not callable: %s') % callback
            LOG.error(msg)
            raise TypeError(tr_msg)
        _SUBSCRIBERS.setdefault(event, {}).setdefault(resource_type, set())
        _SUBSCRIBERS[event][resource_type].add(callback)
        if LOG.logger.getEffectiveLevel() <= log.DEBUG:
            msg = 'Callback: `%(callback)s` subscribed to event `%(event)s`.'
            callback_info = _get_callback_info(callback)
            callback_str = '.'.join((i for i in callback_info if i is not None))
            event_str = '.'.join(['identity', resource_type, event])
            LOG.debug(msg, {'callback': callback_str, 'event': event_str})