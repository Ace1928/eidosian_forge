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
def get_influxdb_client(self):
    influx_args = {}
    if not self._influxdb_config:
        return None
    use_udp = bool(self._influxdb_config.get('use_udp', False))
    port = self._influxdb_config.get('port')
    if use_udp:
        influx_args['use_udp'] = True
    if 'port' in self._influxdb_config:
        if use_udp:
            influx_args['udp_port'] = port
        else:
            influx_args['port'] = port
    for key in ['host', 'username', 'password', 'database', 'timeout']:
        if key in self._influxdb_config:
            influx_args[key] = self._influxdb_config[key]
    if influxdb and influx_args:
        try:
            return influxdb.InfluxDBClient(**influx_args)
        except Exception:
            self.log.warning('Cannot establish connection to InfluxDB')
    else:
        self.log.warning('InfluxDB configuration is present, but no client library is found.')
    return None