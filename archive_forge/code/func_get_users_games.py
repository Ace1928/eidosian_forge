from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def get_users_games(self, steam_id: str) -> List[str]:
    return self.steam.users.get_owned_games(steam_id, False, False)