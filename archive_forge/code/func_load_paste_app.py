import logging
import os
from oslo_config import cfg
from oslo_middleware import cors
from oslo_policy import opts
from oslo_policy import policy
from paste import deploy
from glance.i18n import _
from glance.version import version_info as version
def load_paste_app(app_name, flavor=None, conf_file=None):
    """
    Builds and returns a WSGI app from a paste config file.

    We assume the last config file specified in the supplied ConfigOpts
    object is the paste config file, if conf_file is None.

    :param app_name: name of the application to load
    :param flavor: name of the variant of the application to load
    :param conf_file: path to the paste config file

    :raises RuntimeError: when config file cannot be located or application
            cannot be loaded from config file
    """
    app_name += _get_deployment_flavor(flavor)
    if not conf_file:
        conf_file = _get_deployment_config_file()
    try:
        logger = logging.getLogger(__name__)
        logger.debug('Loading %(app_name)s from %(conf_file)s', {'conf_file': conf_file, 'app_name': app_name})
        app = deploy.loadapp('config:%s' % conf_file, name=app_name)
        if CONF.debug:
            CONF.log_opt_values(logger, logging.DEBUG)
        return app
    except (LookupError, ImportError) as e:
        msg = _('Unable to load %(app_name)s from configuration file %(conf_file)s.\nGot: %(e)r') % {'app_name': app_name, 'conf_file': conf_file, 'e': e}
        logger.error(msg)
        raise RuntimeError(msg)