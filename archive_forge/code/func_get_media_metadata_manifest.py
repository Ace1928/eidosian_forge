import json
import requests
from langchain_core.pydantic_v1 import BaseModel
def get_media_metadata_manifest(self, query: str) -> str:
    response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + '/asset/' + query)
    return response.json()