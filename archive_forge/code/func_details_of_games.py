from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def details_of_games(self, name: str) -> str:
    games = self.steam.apps.search_games(name)
    info_partOne_dict = self.get_id_link_price(games)
    info_partOne = self.parse_to_str(info_partOne_dict)
    id = str(info_partOne_dict.get('id'))
    info_dict = self.steam.apps.get_app_details(id)
    data = info_dict.get(id).get('data')
    detailed_description = data.get('detailed_description')
    detailed_description = self.remove_html_tags(detailed_description)
    supported_languages = info_dict.get(id).get('data').get('supported_languages')
    info_partTwo = 'The summary of the game is: ' + detailed_description + '\n' + 'The supported languages of the game are: ' + supported_languages + '\n'
    info = info_partOne + info_partTwo
    return info