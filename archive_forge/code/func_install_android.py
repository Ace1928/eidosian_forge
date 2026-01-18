def install_android():
    """Install hooks for the android platform.

    * Automatically sleep when the device is paused.
    * Automatically kill the application when the return key is pressed.
    """
    try:
        import android
    except ImportError:
        print('Android lib is missing, cannot install android hooks')
        return
    from kivy.clock import Clock
    from kivy.logger import Logger
    import pygame
    Logger.info('Support: Android install hooks')
    android.init()
    android.map_key(android.KEYCODE_MENU, pygame.K_MENU)
    android.map_key(android.KEYCODE_BACK, pygame.K_ESCAPE)

    def android_check_pause(*largs):
        if not android.check_pause():
            return
        from kivy.app import App
        from kivy.base import stopTouchApp
        from kivy.logger import Logger
        from kivy.core.window import Window
        global g_android_redraw_count, _redraw_event
        Logger.info('Android: Must go into sleep mode, check the app')
        app = App.get_running_app()
        if app is None:
            Logger.info('Android: No app running, stop everything.')
            stopTouchApp()
            return
        if app.dispatch('on_pause'):
            Logger.info('Android: App paused, now wait for resume.')
            android.wait_for_resume()
            if android.check_stop():
                Logger.info('Android: Android wants to close our app.')
                stopTouchApp()
            else:
                Logger.info('Android: Android has resumed, resume the app.')
                app.dispatch('on_resume')
                Window.canvas.ask_update()
                g_android_redraw_count = 25
                if _redraw_event is None:
                    _redraw_event = Clock.schedule_interval(_android_ask_redraw, 1 / 5)
                else:
                    _redraw_event.cancel()
                    _redraw_event()
                Logger.info('Android: App resume completed.')
        else:
            Logger.info("Android: App doesn't support pause mode, stop.")
            stopTouchApp()
    Clock.schedule_interval(android_check_pause, 0)