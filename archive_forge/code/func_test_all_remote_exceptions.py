import inspect
import re
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob
import heat.api.middleware.fault as fault
from heat.common import exception as heat_exc
from heat.common.i18n import _
from heat.tests import common
def test_all_remote_exceptions(self):
    for name, obj in inspect.getmembers(heat_exc, lambda x: inspect.isclass(x) and issubclass(x, heat_exc.HeatException)):
        if '__init__' in obj.__dict__:
            if obj == heat_exc.HeatException:
                continue
            elif obj == heat_exc.Error:
                error = obj('Error')
            elif obj == heat_exc.NotFound:
                error = obj()
            elif obj == heat_exc.ResourceFailure:
                exc = heat_exc.Error(_('Error'))
                error = obj(exc, None, 'CREATE')
            elif obj == heat_exc.ResourcePropertyConflict:
                error = obj('%s' % 'a test prop')
            else:
                continue
            self.remote_exception_helper(name, error)
            continue
        if hasattr(obj, 'msg_fmt'):
            kwargs = {}
            spec_names = re.findall('%\\((\\w+)\\)([cdeEfFgGinorsxX])', obj.msg_fmt)
            for key, convtype in spec_names:
                if convtype == 'r' or convtype == 's':
                    kwargs[key] = '"' + key + '"'
                else:
                    raise Exception("test needs additional conversion type added due to %s exception using '%c' specifier" % (obj, convtype))
            error = obj(**kwargs)
            self.remote_exception_helper(name, error)