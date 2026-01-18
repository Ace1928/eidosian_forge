import copy
from oslo_serialization import jsonutils
import yaml
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import signal_responder
from heat.engine import support
from heat.engine import translation
def parse_execution_response(execution):
    return {'id': execution.id, 'workflow_name': execution.workflow_name, 'created_at': execution.created_at, 'updated_at': execution.updated_at, 'state': execution.state, 'input': jsonutils.loads(str(execution.input)), 'output': jsonutils.loads(str(execution.output))}