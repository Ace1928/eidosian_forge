from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def get_steam_id(self, name: str) -> str:
    user = self.steam.users.search_user(name)
    steam_id = user['player']['steamid']
    return steam_id