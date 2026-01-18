import collections.abc
import functools
import itertools
import threading
import typing as ty
import uuid
import warnings
import debtcollector
from debtcollector import renames
def get_logging_values(self) -> ty.Dict[str, ty.Any]:
    """Return a dictionary of logging specific context attributes."""
    values = {'user_name': self.user_name, 'project_name': self.project_name, 'domain_name': self.domain_name, 'user_domain_name': self.user_domain_name, 'project_domain_name': self.project_domain_name}
    values.update(self.to_dict())
    if self.auth_token:
        values['auth_token'] = '***'
    else:
        values['auth_token'] = None
    values.pop('auth_token_info', None)
    return values