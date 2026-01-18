from abc import ABC, abstractmethod
import copy
from threading import Lock
from typing import Dict, Iterable, List, Optional
from .metrics_core import Metric
def set_target_info(self, labels: Optional[Dict[str, str]]) -> None:
    with self._lock:
        if labels:
            if not self._target_info and 'target_info' in self._names_to_collectors:
                raise ValueError('CollectorRegistry already contains a target_info metric')
            self._names_to_collectors['target_info'] = _EmptyCollector()
        elif self._target_info:
            self._names_to_collectors.pop('target_info', None)
        self._target_info = labels