import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
def opts_to_ini(uri, *args, **kwargs):
    opts = _extra_configs[uri]['data']
    result = ''
    for g in opts.keys():
        result += '[{}]\n'.format(g)
        for o, (t, v) in opts[g].items():
            if t == cfg.MultiStrOpt:
                for i in v:
                    result += '{} = {}\n'.format(o, i)
            else:
                result += '{} = {}\n'.format(o, v)
    return result