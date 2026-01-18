from sqlalchemy.orm import exc
import tenacity
def retry_on_stale_data_error(func):
    wrapper = tenacity.retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(exc.StaleDataError), reraise=True)
    return wrapper(func)