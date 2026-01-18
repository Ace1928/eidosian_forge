from botocore.compat import six
from s3transfer.compat import accepts_kwargs
from s3transfer.exceptions import InvalidSubscriberMethodError
def on_done(self, future, **kwargs):
    """Callback to be invoked once a transfer is done

        This callback can be useful for:

            * Recording and displaying whether the transfer succeeded or
              failed using future.result()
            * Running some task after the transfer completed like changing
              the last modified time of a downloaded file.

        :type future: s3transfer.futures.TransferFuture
        :param future: The TransferFuture representing the requested transfer.
        """
    pass