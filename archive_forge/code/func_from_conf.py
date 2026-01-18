import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
def from_conf(conf, session=None, service_types=None, **kwargs):
    """Create a CloudRegion from oslo.config ConfigOpts.

    :param oslo_config.cfg.ConfigOpts conf:
        An oslo.config ConfigOpts containing keystoneauth1.Adapter options in
        sections named according to project (e.g. [nova], not [compute]).
        TODO: Current behavior is to use defaults if no such section exists,
        which may not be what we want long term.
    :param keystoneauth1.session.Session session:
        An existing authenticated Session to use. This is currently required.
        TODO: Load this (and auth) from the conf.
    :param service_types:
        A list/set of service types for which to look for and process config
        opts. If None, all known service types are processed. Note that we will
        not error if a supplied service type can not be processed successfully
        (unless you try to use the proxy, of course). This tolerates uses where
        the consuming code has paths for a given service, but those paths are
        not exercised for given end user setups, and we do not want to generate
        errors for e.g. missing/invalid conf sections in those cases. We also
        don't check to make sure your service types are spelled correctly -
        caveat implementor.
    :param kwargs:
        Additional keyword arguments to be passed directly to the CloudRegion
        constructor.
    :raise openstack.exceptions.ConfigException:
        If session is not specified.
    :return:
        An openstack.config.cloud_region.CloudRegion.
    """
    if not session:
        raise exceptions.ConfigException('A Session must be supplied.')
    config_dict = kwargs.pop('config', config_defaults.get_defaults())
    stm = os_service_types.ServiceTypes()
    for st in stm.all_types_by_service_type:
        if service_types is not None and st not in service_types:
            _disable_service(config_dict, st, reason='Not in the list of requested service_types.')
            continue
        project_name = stm.get_project_name(st)
        if project_name not in conf:
            if '-' in project_name:
                project_name = project_name.replace('-', '_')
            if project_name not in conf:
                _disable_service(config_dict, st, reason="No section for project '{project}' (service type '{service_type}') was present in the config.".format(project=project_name, service_type=st))
                continue
        opt_dict: ty.Dict[str, str] = {}
        try:
            ks_load_adap.process_conf_options(conf[project_name], opt_dict)
        except Exception as e:
            reason = "Encountered an exception attempting to process config for project '{project}' (service type '{service_type}'): {exception}".format(project=project_name, service_type=st, exception=e)
            _logger.warning("Disabling service '{service_type}': {reason}".format(service_type=st, reason=reason))
            _disable_service(config_dict, st, reason=reason)
            continue
        for raw_name, opt_val in opt_dict.items():
            config_name = _make_key(raw_name, st)
            config_dict[config_name] = opt_val
    return CloudRegion(session=session, config=config_dict, **kwargs)