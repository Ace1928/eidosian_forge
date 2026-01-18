import os
from .start_hook_base import RayOnSparkStartHook
from .utils import get_spark_session
import logging
import threading
import time
def display_databricks_driver_proxy_url(spark_context, port, title):
    """
    This helper function create a proxy URL for databricks driver webapp forwarding.
    In databricks runtime, user does not have permission to directly access web
    service binding on driver machine port, but user can visit it by a proxy URL with
    following format: "/driver-proxy/o/{orgId}/{clusterId}/{port}/".
    """
    from dbruntime.display import displayHTML
    driverLocal = spark_context._jvm.com.databricks.backend.daemon.driver.DriverLocal
    commandContextTags = driverLocal.commandContext().get().toStringMap().apply('tags')
    orgId = commandContextTags.apply('orgId')
    clusterId = commandContextTags.apply('clusterId')
    proxy_link = f'/driver-proxy/o/{orgId}/{clusterId}/{port}/'
    proxy_url = f'https://dbc-dp-{orgId}.cloud.databricks.com{proxy_link}'
    print('To monitor and debug Ray from Databricks, view the dashboard at ')
    print(f' {proxy_url}')
    displayHTML(f'\n      <div style="margin-bottom: 16px">\n          <a href="{proxy_link}">\n              Open {title} in a new tab\n          </a>\n      </div>\n    ')