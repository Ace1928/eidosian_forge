import warnings
from typing import Any, Dict, List, Optional, Callable, Tuple
from mypy_extensions import Arg, KwArg
from langchain_community.tools.file_management import ReadFileTool
from langchain_core.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.callbacks import Callbacks
from langchain.chains.api import news_docs, open_meteo_docs, podcast_docs, tmdb_docs
from langchain.chains.api.base import APIChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.golden_query.tool import GoldenQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.tools import BaseTool
from langchain_community.tools.bing_search.tool import BingSearchRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.google_cloud.texttospeech import (
from langchain_community.tools.google_lens.tool import GoogleLensQueryRun
from langchain_community.tools.google_search.tool import (
from langchain_community.tools.google_scholar.tool import GoogleScholarQueryRun
from langchain_community.tools.google_finance.tool import GoogleFinanceQueryRun
from langchain_community.tools.google_trends.tool import GoogleTrendsQueryRun
from langchain_community.tools.metaphor_search.tool import MetaphorSearchResults
from langchain_community.tools.google_jobs.tool import GoogleJobsQueryRun
from langchain_community.tools.google_serper.tool import (
from langchain_community.tools.searchapi.tool import SearchAPIResults, SearchAPIRun
from langchain_community.tools.graphql.tool import BaseGraphQLTool
from langchain_community.tools.human.tool import HumanInputRun
from langchain_community.tools.requests.tool import (
from langchain_community.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool
from langchain_community.tools.scenexplain.tool import SceneXplainTool
from langchain_community.tools.searx_search.tool import (
from langchain_community.tools.shell.tool import ShellTool
from langchain_community.tools.sleep.tool import SleepTool
from langchain_community.tools.stackexchange.tool import StackExchangeTool
from langchain_community.tools.merriam_webster.tool import MerriamWebsterQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.tools.dataforseo_api_search import DataForSeoAPISearchRun
from langchain_community.tools.dataforseo_api_search import DataForSeoAPISearchResults
from langchain_community.tools.memorize.tool import Memorize
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_community.utilities.metaphor_search import MetaphorSearchAPIWrapper
from langchain_community.utilities.awslambda import LambdaWrapper
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_community.utilities.searchapi import SearchApiAPIWrapper
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.stackexchange import StackExchangeAPIWrapper
from langchain_community.utilities.twilio import TwilioAPIWrapper
from langchain_community.utilities.merriam_webster import MerriamWebsterAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_community.utilities.dataforseo_api_search import DataForSeoAPIWrapper
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
def load_tools(tool_names: List[str], llm: Optional[BaseLanguageModel]=None, callbacks: Callbacks=None, allow_dangerous_tools: bool=False, **kwargs: Any) -> List[BaseTool]:
    """Load tools based on their name.

    Tools allow agents to interact with various resources and services like
    APIs, databases, file systems, etc.

    Please scope the permissions of each tools to the minimum required for the
    application.

    For example, if an application only needs to read from a database,
    the database tool should not be given write permissions. Moreover
    consider scoping the permissions to only allow accessing specific
    tables and impose user-level quota for limiting resource usage.

    Please read the APIs of the individual tools to determine which configuration
    they support.

    See [Security](https://python.langchain.com/docs/security) for more information.

    Args:
        tool_names: name of tools to load.
        llm: An optional language model, may be needed to initialize certain tools.
        callbacks: Optional callback manager or list of callback handlers.
            If not provided, default global callback manager will be used.
        allow_dangerous_tools: Optional flag to allow dangerous tools.
            Tools that contain some level of risk.
            Please use with caution and read the documentation of these tools
            to understand the risks and how to mitigate them.
            Refer to https://python.langchain.com/docs/security
            for more information.
            Please note that this list may not be fully exhaustive.
            It is your responsibility to understand which tools
            you're using and the risks associated with them.

    Returns:
        List of tools.
    """
    tools = []
    callbacks = _handle_callbacks(callback_manager=kwargs.get('callback_manager'), callbacks=callbacks)
    for name in tool_names:
        if name in DANGEROUS_TOOLS and (not allow_dangerous_tools):
            raise ValueError(f"""{name} is a dangerous tool. You cannot use it without opting in by setting allow_dangerous_tools to True. Most tools have some inherit risk to them merely because they are allowed to interact with the "real world".Please refer to LangChain security guidelines to https://python.langchain.com/docs/security.Some tools have been designated as dangerous because they pose risk that is not intuitively obvious. For example, a tool that allows an agent to make requests to the web, can also be used to make requests to a server that is only accessible from the server hosting the code.Again, all tools carry some risk, and it's your responsibility to understand which tools you're using and the risks associated with them.""")
        if name in {'requests'}:
            warnings.warn('tool name `requests` is deprecated - please use `requests_all` or specify the requests method')
        if name == 'requests_all':
            requests_method_tools = [_tool for _tool in _BASE_TOOLS if _tool.startswith('requests_')]
            tool_names.extend(requests_method_tools)
        elif name in _BASE_TOOLS:
            tools.append(_BASE_TOOLS[name]())
        elif name in DANGEROUS_TOOLS:
            tools.append(DANGEROUS_TOOLS[name]())
        elif name in _LLM_TOOLS:
            if llm is None:
                raise ValueError(f'Tool {name} requires an LLM to be provided')
            tool = _LLM_TOOLS[name](llm)
            tools.append(tool)
        elif name in _EXTRA_LLM_TOOLS:
            if llm is None:
                raise ValueError(f'Tool {name} requires an LLM to be provided')
            _get_llm_tool_func, extra_keys = _EXTRA_LLM_TOOLS[name]
            missing_keys = set(extra_keys).difference(kwargs)
            if missing_keys:
                raise ValueError(f'Tool {name} requires some parameters that were not provided: {missing_keys}')
            sub_kwargs = {k: kwargs[k] for k in extra_keys}
            tool = _get_llm_tool_func(llm=llm, **sub_kwargs)
            tools.append(tool)
        elif name in _EXTRA_OPTIONAL_TOOLS:
            _get_tool_func, extra_keys = _EXTRA_OPTIONAL_TOOLS[name]
            sub_kwargs = {k: kwargs[k] for k in extra_keys if k in kwargs}
            tool = _get_tool_func(**sub_kwargs)
            tools.append(tool)
        else:
            raise ValueError(f'Got unknown tool {name}')
    if callbacks is not None:
        for tool in tools:
            tool.callbacks = callbacks
    return tools