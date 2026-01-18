import collections
import dataclasses
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
def neuron_monitor(self) -> None:
    """Run neuron-monitor in a separate process to collect raw data."""
    self.write_neuron_monitor_config()
    try:
        command = [NEURON_MONITOR_PATH, '-c', self.neuron_monitor_config_path]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None) as process:
            while not self.shutdown_event.is_set():
                if process.stdout is None:
                    self.shutdown_event.wait(0.1)
                    continue
                raw_data = process.stdout.readline()
                if raw_data:
                    self.raw_samples.append(raw_data)
            process.kill()
            process.wait()
    except Exception as e:
        logger.error('neuron-monitor failed: %s' % e)