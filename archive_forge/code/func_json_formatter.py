import numbers
import prettytable
import yaml
from osc_lib import exceptions as exc
from oslo_serialization import jsonutils
def json_formatter(js):
    formatter = jsonutils.dumps(js, indent=2, ensure_ascii=False)
    return formatter