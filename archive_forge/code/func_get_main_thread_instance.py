def get_main_thread_instance(threading):
    if hasattr(threading, 'main_thread'):
        return threading.main_thread()
    else:
        return threading._shutdown.im_self