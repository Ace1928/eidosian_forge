from typing import List
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.polygon import (
from langchain_community.utilities.polygon import PolygonAPIWrapper
@classmethod
def from_polygon_api_wrapper(cls, polygon_api_wrapper: PolygonAPIWrapper) -> 'PolygonToolkit':
    tools = [PolygonAggregates(api_wrapper=polygon_api_wrapper), PolygonLastQuote(api_wrapper=polygon_api_wrapper), PolygonTickerNews(api_wrapper=polygon_api_wrapper), PolygonFinancials(api_wrapper=polygon_api_wrapper)]
    return cls(tools=tools)