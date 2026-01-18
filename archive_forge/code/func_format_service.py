import uuid
from oslo_utils import timeutils
from heat.rpc import listener_client
def format_service(service):
    if service is None:
        return
    status = 'down'
    last_updated = service.updated_at or service.created_at
    check_interval = (timeutils.utcnow() - last_updated).total_seconds()
    if check_interval <= 2 * service.report_interval:
        status = 'up'
    result = {SERVICE_ID: service.id, SERVICE_BINARY: service.binary, SERVICE_ENGINE_ID: service.engine_id, SERVICE_HOST: service.host, SERVICE_HOSTNAME: service.hostname, SERVICE_TOPIC: service.topic, SERVICE_REPORT_INTERVAL: service.report_interval, SERVICE_CREATED_AT: service.created_at, SERVICE_UPDATED_AT: service.updated_at, SERVICE_DELETED_AT: service.deleted_at, SERVICE_STATUS: status}
    return result