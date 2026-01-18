from cmakelang.parse.additional_nodes import ShellCommandNode
from cmakelang.parse.argument_nodes import (
def parse_fetchcontent_populate(ctx, tokens, breakstack):
    """
  ::

    FetchContent_Populate( <name>
      [QUIET]
      [SUBBUILD_DIR <subBuildDir>]
      [SOURCE_DIR <srcDir>]
      [BINARY_DIR <binDir>]
      ...
    )

  :see: https://cmake.org/cmake/help/v3.14/module/FetchContent.html?highlight=fetchcontent#command:fetchcontent_populate
  """
    return StandardArgTree.parse(ctx, tokens, npargs=1, kwargs={'SUBBUILD_DIR': PositionalParser(1), 'SOURCE_DIR': PositionalParser(1), 'BINARY_DIR': PositionalParser(1), 'DOWNLOAD_COMMAND': ShellCommandNode.parse, 'URL': PositionalParser('+'), 'URL_HASH': PositionalParser(1), 'URL_MD5': PositionalParser(1), 'DOWNLOAD_NAME': PositionalParser(1), 'DOWNLOAD_NO_EXTRACT': PositionalParser(1), 'DOWNLOAD_NO_PROGRESS': PositionalParser(1), 'TIMEOUT': PositionalParser(1), 'HTTP_USERNAME': PositionalParser(1), 'HTTP_PASSWORD': PositionalParser(1), 'HTTP_HEADER': PositionalParser('+'), 'TLS_VERIFY': PositionalParser(1), 'TLS_CAINFO': PositionalParser(1), 'NETRC': PositionalParser(1, flags=['CMAKE_NETRC', 'IGNORED', 'OPTIONAL', 'REQUIRED']), 'NETRC_FILE': PositionalParser(1), 'GIT_REPOSITORY': PositionalParser(1), 'GIT_TAG': PositionalParser(1), 'GIT_REMOTE_NAME': PositionalParser(1), 'GIT_SUBMODULES': PositionalParser('+'), 'GIT_SHALLOW': PositionalParser(1), 'GIT_PROGRESS': PositionalParser(1), 'GIT_CONFIG': PositionalParser('+'), 'SVN_REPOSITORY': PositionalParser(1), 'SVN_REVISION': PositionalParser(1), 'SVN_USERNAME': PositionalParser(1), 'SVN_PASSWORD': PositionalParser(1), 'SVN_TRUST_CERT': PositionalParser(1), 'HG_REPOSITORY': PositionalParser(1), 'HG_TAG': PositionalParser(1), 'CVS_REPOSITORY': PositionalParser(1), 'CVS_MODULE': PositionalParser(1), 'CVS_TAG': PositionalParser(1), 'UPDATE_COMMAND': ShellCommandNode.parse, 'UPDATE_DISCONNECTED': PositionalParser(1), 'PATCH_COMMAND': ShellCommandNode.parse}, flags=['QUIET'], breakstack=breakstack)