from bs4 import BeautifulSoup
@property
def redux_state_paths(self):
    return self.driver.execute_script('\n            var p = window.store.getState().paths;\n            return {strs: p.strs, objs: p.objs}\n            ')