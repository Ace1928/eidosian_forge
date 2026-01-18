import asyncio
def debounced(*args, **kwargs):
    nonlocal timer

    def call_it():
        fn(*args, **kwargs)
    if timer is not None:
        timer.cancel()
    timer = Timer(wait, call_it)