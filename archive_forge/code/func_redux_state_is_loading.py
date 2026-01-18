from bs4 import BeautifulSoup
@property
def redux_state_is_loading(self):
    return self.driver.execute_script('\n            return window.store.getState().isLoading;\n            ')