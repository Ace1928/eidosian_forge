import os
import random
from selenium import webdriver
def get_chrome(proxy=None, useragent='random'):
    _driver = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chromedriver')
    _options = webdriver.chrome.options.Options()
    _options.add_argument('--headless')
    _options.add_argument('--no-sandbox')
    _options.add_argument('--disable-notifications')
    if useragent == 'random':
        _options.add_argument(f'user-agent={random.choice(uagents)}')
    elif useragent == 'custom':
        pass
    else:
        _options.add_argument(f'user-agent={useragent}')
    if proxy:
        _options.add_argument('--proxy-server=socks5://{}'.format(proxy))
    browser = webdriver.Chrome(options=_options, executable_path=_driver)
    browser.set_page_load_timeout(60)
    browser.set_window_size(1366, 768)
    return browser