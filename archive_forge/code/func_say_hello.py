from ray.util.client import ray
from typing import Tuple
def say_hello(self, whom: str) -> Tuple[str, int]:
    self.count += 1
    return ('Hello ' + whom, self.count)