from lazyops.types.common import UpperStrEnum
from typing import Union
@classmethod
def parse_role(cls, role: Union[str, int]) -> 'UserRole':
    """
        Parses the role
        """
    if role is None:
        return UserRole.ANON
    return cls(UserPrivilageIntLevel[role]) if isinstance(role, int) else cls(role.upper())