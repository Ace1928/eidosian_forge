import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def get_all_unique_module_fqns(self) -> Set[str]:
    """
        The purpose of this method is to provide a user the set of all module_fqns so that if
        they wish to use some of the filtering capabilities of the ModelReportVisualizer class,
        they don't need to manually parse the generated_reports dictionary to get this information.

        Returns all the unique module fqns present in the reports the ModelReportVisualizer
        instance was initialized with.
        """
    return set(self.generated_reports.keys())