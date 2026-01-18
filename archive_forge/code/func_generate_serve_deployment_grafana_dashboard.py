import copy
from dataclasses import asdict
import json
import os
from typing import List, Tuple
import ray
from ray.dashboard.modules.metrics.dashboards.common import DashboardConfig, Panel
from ray.dashboard.modules.metrics.dashboards.default_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.serve_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.serve_deployment_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.data_dashboard_panels import (
def generate_serve_deployment_grafana_dashboard() -> Tuple[str, str]:
    """
    Generates the dashboard output for the serve dashboard and returns
    both the content and the uid.

    Returns:
      Tuple with format content, uid
    """
    return _generate_grafana_dashboard(serve_deployment_dashboard_config)