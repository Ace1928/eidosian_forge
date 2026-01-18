from heat.db import api as db_api
from heat.engine import service
from heat.engine import stack
from heat.tests.convergence.framework import message_processor
from heat.tests.convergence.framework import message_queue
from heat.tests.convergence.framework import scenario_template
from heat.tests import utils
@message_processor.asynchronous
def rollback_stack(self, stack_name):
    cntxt = utils.dummy_context()
    db_stack = db_api.stack_get_by_name(cntxt, stack_name)
    stk = stack.Stack.load(cntxt, stack=db_stack)
    stk.thread_group_mgr = SynchronousThreadGroupManager()
    stk.rollback()