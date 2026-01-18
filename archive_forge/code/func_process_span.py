from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def process_span(self, run: Run) -> Optional['Span']:
    """Converts a LangChain Run into a W&B Trace Span.
        :param run: The LangChain Run to convert.
        :return: The converted W&B Trace Span.
        """
    try:
        span = self._convert_lc_run_to_wb_span(run)
        return span
    except Exception as e:
        if PRINT_WARNINGS:
            self.wandb.termwarn(f'Skipping trace saving - unable to safely convert LangChain Run into W&B Trace due to: {e}')
        return None