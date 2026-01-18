import json
import requests
from langchain_core.pydantic_v1 import BaseModel
def get_video_captions_location(self, query: str) -> str:
    response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + '/captions/' + query)
    return response.json()