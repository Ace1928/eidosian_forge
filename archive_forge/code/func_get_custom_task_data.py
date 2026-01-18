from parlai.core.worlds import World
from parlai.mturk.core.dev.agents import AssignState
def get_custom_task_data(self):
    """
        This function should take the contents of whatever was collected during this
        task that should be saved and return it in some format, preferrably a dict
        containing acts.

        If data needs pickling, put it in a field named 'needs-pickle'
        """
    pass