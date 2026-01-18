import contextlib
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.common import service_utils
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
def get_engine_id(self):
    """Return the ID of the engine which currently holds the lock.

        Returns None if there is no lock held on the stack.
        """
    return stack_lock_object.StackLock.get_engine_id(self.context, self.stack_id)