from twisted.web import resource
def rewriter(request):
    if request.postpath[:len(aliasPath)] == aliasPath:
        after = request.postpath[len(aliasPath):]
        request.postpath = sourcePath + after
        request.path = '/' + '/'.join(request.prepath + request.postpath)