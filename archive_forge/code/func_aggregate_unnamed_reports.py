import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
from typing import Union, List, Optional, Tuple, Set, Any, Dict
import torch
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector
def aggregate_unnamed_reports(reports: List[Dict[str, Metric]]) -> Dict[str, Metric]:
    """
    Combines metrics without regard for tracking provenence.
    """
    m: Dict[str, Metric] = {}
    for task_report in reports:
        for each_metric, value in task_report.items():
            m[each_metric] = m.get(each_metric) + value
    return m