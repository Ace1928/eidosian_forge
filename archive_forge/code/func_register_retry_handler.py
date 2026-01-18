import logging
import math
import threading
from botocore.retries import bucket, standard, throttling
def register_retry_handler(client):
    clock = bucket.Clock()
    rate_adjustor = throttling.CubicCalculator(starting_max_rate=0, start_time=clock.current_time())
    token_bucket = bucket.TokenBucket(max_rate=1, clock=clock)
    rate_clocker = RateClocker(clock)
    throttling_detector = standard.ThrottlingErrorDetector(retry_event_adapter=standard.RetryEventAdapter())
    limiter = ClientRateLimiter(rate_adjustor=rate_adjustor, rate_clocker=rate_clocker, token_bucket=token_bucket, throttling_detector=throttling_detector, clock=clock)
    client.meta.events.register('before-send', limiter.on_sending_request)
    client.meta.events.register('needs-retry', limiter.on_receiving_response)
    return limiter