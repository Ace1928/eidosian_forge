from ray.tune.search.search_algorithm import SearchAlgorithm
from ray.tune.search.searcher import Searcher
from ray.tune.search.concurrency_limiter import ConcurrencyLimiter
from ray.tune.search.repeater import Repeater
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.variant_generator import grid_search
from ray.tune.search.search_generator import SearchGenerator
from ray._private.utils import get_function_args
from ray.util import PublicAPI
@PublicAPI(stability='beta')
def create_searcher(search_alg, **kwargs):
    """Instantiate a search algorithm based on the given string.

    This is useful for swapping between different search algorithms.

    Args:
        search_alg: The search algorithm to use.
        metric: The training result objective value attribute. Stopping
            procedures will use this attribute.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        **kwargs: Additional parameters.
            These keyword arguments will be passed to the initialization
            function of the chosen class.
    Returns:
        ray.tune.search.Searcher: The search algorithm.
    Example:
        >>> from ray import tune # doctest: +SKIP
        >>> search_alg = tune.create_searcher('ax') # doctest: +SKIP
    """
    search_alg = search_alg.lower()
    if search_alg not in SEARCH_ALG_IMPORT:
        raise ValueError(f'The `search_alg` argument must be one of {list(SEARCH_ALG_IMPORT)}. Got: {search_alg}')
    SearcherClass = SEARCH_ALG_IMPORT[search_alg]()
    search_alg_args = get_function_args(SearcherClass)
    trimmed_kwargs = {k: v for k, v in kwargs.items() if k in search_alg_args}
    return SearcherClass(**trimmed_kwargs)