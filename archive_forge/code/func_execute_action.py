import enum
import threading
from cupyx.distributed import _klv_utils
def execute_action(action, value, store):
    try:
        if action == Actions.Set:
            action_obj = Set.from_klv(value)
        elif action == Actions.Get:
            action_obj = Get.from_klv(value)
        elif action == Actions.Barrier:
            action_obj = Barrier.from_klv(value)
        else:
            raise ValueError(f'unknown action {action}')
        return action_obj(store)
    except Exception as e:
        return ActionError(e)