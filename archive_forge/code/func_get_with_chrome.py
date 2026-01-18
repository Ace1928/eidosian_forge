import os
import random
from selenium import webdriver
def get_with_chrome(url, proxy=None, useragent='random'):
    browser = get_chrome(proxy, useragent)
    try:
        browser.get(url)
        code = browser.page_source
    except:
        code = ''
    finally:
        browser.stop_client()
        browser.close()
        browser.quit()
        del browser
    return code