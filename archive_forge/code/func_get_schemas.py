from __future__ import annotations
import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union
import aiohttp
import requests
from aiohttp import ServerTimeoutError
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from requests.exceptions import Timeout
def get_schemas(self) -> str:
    """Get the available schema's."""
    if self.schemas:
        return ', '.join([f'{key}: {value}' for key, value in self.schemas.items()])
    return "No known schema's yet. Use the schema_powerbi tool first."