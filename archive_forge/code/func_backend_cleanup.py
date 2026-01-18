from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@app.task(name='celery.backend_cleanup', shared=False, lazy=False)
def backend_cleanup():
    app.backend.cleanup()