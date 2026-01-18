import logging
import torch
import pandas as pd
import concurrent.futures
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Dict, Any, Optional
import sys
import os
import importlib.util
def parallelized_computation(self, activation_dict):
    """
        Executes the initialization of activation functions in a parallelized manner using a ThreadPoolExecutor.
        """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_activation = {executor.submit(lambda x: activation_dict[x](), name): name for name in activation_dict}
        for future in concurrent.futures.as_completed(future_to_activation):
            activation_name = future_to_activation[future]
            try:
                data = future.result()
            except Exception as exc:
                logging.error(f'{activation_name} generated an exception: {exc}')
            else:
                logging.info(f'{activation_name}: Computation result - {data}')
    for func_name, func in self.activation_types.items():
        logging.debug(f'Initialized {func_name} activation function with lambda expression: {func}')
    parallelized_computation(self.activation_types)