from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
def call_actor(self, actor_id: str, run_input: Dict, dataset_mapping_function: Callable[[Dict], Document], *, build: Optional[str]=None, memory_mbytes: Optional[int]=None, timeout_secs: Optional[int]=None) -> 'ApifyDatasetLoader':
    """Run an Actor on the Apify platform and wait for results to be ready.
        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            run_input (Dict): The input object of the Actor that you're trying to run.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                Actor run's default dataset.
        """
    from langchain_community.document_loaders import ApifyDatasetLoader
    actor_call = self.apify_client.actor(actor_id).call(run_input=run_input, build=build, memory_mbytes=memory_mbytes, timeout_secs=timeout_secs)
    return ApifyDatasetLoader(dataset_id=actor_call['defaultDatasetId'], dataset_mapping_function=dataset_mapping_function)