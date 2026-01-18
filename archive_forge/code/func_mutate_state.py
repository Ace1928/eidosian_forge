import re
from django.db.migrations.utils import get_migration_name_timestamp
from django.db.transaction import atomic
from .exceptions import IrreversibleError
def mutate_state(self, project_state, preserve=True):
    """
        Take a ProjectState and return a new one with the migration's
        operations applied to it. Preserve the original object state by
        default and return a mutated state from a copy.
        """
    new_state = project_state
    if preserve:
        new_state = project_state.clone()
    for operation in self.operations:
        operation.state_forwards(self.app_label, new_state)
    return new_state