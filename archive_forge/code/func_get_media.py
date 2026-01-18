import json
import requests
from langchain_core.pydantic_v1 import BaseModel
def get_media(self, query: str) -> str:
    params = json.loads(query)
    if params.get('q'):
        queryText = params['q']
        params.pop('q')
    else:
        queryText = ''
    response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + '/search?q=' + queryText, params=params)
    data = response.json()
    return data